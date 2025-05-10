from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import os
import openai

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gemini-1.5-pro_batch",
    "gemini-1.5-pro",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "disable_translation",
]
DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gemini-1.5-pro",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    """
    Translate text segments individually to the specified language.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for
        segment text.
    - target (str): Target language code.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Translates each segment using Google Translate.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated_segments = translate_iterative(segments, 'es')
    """

    segments_ = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    translator = GoogleTranslator(source=source, target=target)

    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translator.translate(text.strip())
        segments_[line]["text"] = translated_line

    return segments_


def verify_translate(
    segments,
    segments_copy,
    translated_lines,
    target,
    source
):
    """
    Verify integrity and translate segments if lengths match, otherwise
    switch to iterative translation.
    """
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            logger.debug(
                f"{segments_copy[line]['text']} >> "
                f"{translated_lines[line].strip()}"
            )
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        logger.error(
            "The translation failed, switching to google_translate iterative. "
            f"{len(segments), len(translated_lines)}"
        )
        return translate_iterative(segments, target, source)


def translate_batch(segments, target, chunk_size=2000, source=None):
    """
    Translate a batch of text segments into the specified language in chunks,
        respecting the character limit.

    Parameters:
    - segments (list): List of dictionaries with 'text' as a key for segment
        text.
    - target (str): Target language code.
    - chunk_size (int, optional): Maximum character limit for each translation
        chunk (default is 2000; max 5000).
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Splits input segments into chunks respecting the character limit for
        translation.
    - Translates the chunks using Google Translate.
    - If chunked translation fails, switches to iterative translation using
        `translate_iterative()`.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated = translate_batch(segments, 'es', chunk_size=4000, source='en')
    """

    segments_copy = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    # Get text
    text_lines = []
    for line in range(len(segments_copy)):
        text = segments_copy[line]["text"].strip()
        text_lines.append(text)

    # chunk limit
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    # translate chunks
    progress_bar = tqdm(total=len(segments), desc="Translating")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                logger.debug(
                    "Chunk fixing iteratively. Len chunk: "
                    f"{len(split_text)}, expected: {len(text_iterable)}"
                )
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    split_text.append(translated_txt)
                    progress_bar.update(1)
            split_list.append(split_text)
        progress_bar.close()
    except Exception as error:
        progress_bar.close()
        logger.error(str(error))
        logger.warning(
            "The translation in chunks failed, switching to iterative."
            " Related: too many request"
        )  # use proxy or less chunk size
        return translate_iterative(segments, target, source)

    # un chunk
    translated_lines = list(chain.from_iterable(split_list))

    return verify_translate(
        segments, segments_copy, translated_lines, target, source
    )


def call_gpt_translate(
    client,
    model,
    system_prompt,
    user_prompt,
    original_text=None,
    batch_lines=None,
):

    # https://platform.openai.com/docs/guides/text-generation/json-mode
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    logger.debug(f"Result: {str(result)}")

    try:
        translation = json.loads(result)
    except Exception as error:
        match_result = re.search(r'\{.*?\}', result)
        if match_result:
            logger.error(str(error))
            json_str = match_result.group(0)
            translation = json.loads(json_str)
        else:
            raise error

    # Get valid data
    if batch_lines:
        for conversation in translation.values():
            if isinstance(conversation, dict):
                conversation = list(conversation.values())[0]
            if (
                list(
                    original_text["conversation"][0].values()
                )[0].strip() ==
                list(conversation[0].values())[0].strip()
            ):
                continue
            if len(conversation) == batch_lines:
                break

        fix_conversation_length = []
        for line in conversation:
            for speaker_code, text_tr in line.items():
                fix_conversation_length.append({speaker_code: text_tr})

        logger.debug(f"Data batch: {str(fix_conversation_length)}")
        logger.debug(
            f"Lines Received: {len(fix_conversation_length)},"
            f" expected: {batch_lines}"
        )

        return fix_conversation_length

    else:
        if isinstance(translation, dict):
            translation = list(translation.values())[0]
        if isinstance(translation, list):
            translation = translation[0]
        if isinstance(translation, set):
            translation = list(translation)[0]
        if not isinstance(translation, str):
            raise ValueError(f"No valid response received: {str(translation)}")

        return translation


def gpt_sequential(segments, model, target, source=None):
    from openai import OpenAI

    translated_segments = copy.deepcopy(segments)

    client = OpenAI()
    progress_bar = tqdm(total=len(segments), desc="Translating")

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    system_prompt = "Machine translation designed to output the translated_text JSON."

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        start = line["start"]
        user_prompt = f"Translate the following {lang_sc} text into {lang_tg}, write the fully translated text and nothing more:\n{text}"

        time.sleep(0.5)

        try:
            translated_text = call_gpt_translate(
                client,
                model,
                system_prompt,
                user_prompt,
            )

        except Exception as error:
            logger.error(
                f"{str(error)} >> The text of segment {start} "
                "is being corrected with Google Translate"
            )
            translator = GoogleTranslator(
                source=fixed_source, target=fixed_target
            )
            translated_text = translator.translate(text.strip())

        translated_segments[i]["text"] = translated_text.strip()
        progress_bar.update(1)

    progress_bar.close()

    return translated_segments


def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    from openai import OpenAI
    import tiktoken

    token_batch_limit = max(100, (token_batch_limit - 40) // 2)
    progress_bar = tqdm(total=len(segments), desc="Translating")
    segments_copy = copy.deepcopy(segments)
    encoding = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    name_speaker = "ABCDEFGHIJKL"

    translated_lines = []
    text_data_dict = []
    num_tokens = 0
    count_sk = {char: 0 for char in "ABCDEFGHIJKL"}

    for i, line in enumerate(segments_copy):
        text = line["text"]
        speaker = line["speaker"]
        last_start = line["start"]
        # text_data_dict.append({str(int(speaker[-1])+1): text})
        index_sk = int(speaker[-2:])
        character_sk = name_speaker[index_sk]
        count_sk[character_sk] += 1
        code_sk = character_sk+str(count_sk[character_sk])
        text_data_dict.append({code_sk: text})
        num_tokens += len(encoding.encode(text)) + 7
        if num_tokens >= token_batch_limit or i == len(segments_copy)-1:
            try:
                batch_lines = len(text_data_dict)
                batch_conversation = {"conversation": copy.deepcopy(text_data_dict)}
                # Reset vars
                num_tokens = 0
                text_data_dict = []
                count_sk = {char: 0 for char in "ABCDEFGHIJKL"}
                # Process translation
                # https://arxiv.org/pdf/2309.03409.pdf
                system_prompt = f"Machine translation designed to output the translated_conversation key JSON containing a list of {batch_lines} items."
                user_prompt = f"Translate each of the following text values in conversation{' from' if lang_sc else ''} {lang_sc} to {lang_tg}:\n{batch_conversation}"
                logger.debug(f"Prompt: {str(user_prompt)}")

                conversation = call_gpt_translate(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    original_text=batch_conversation,
                    batch_lines=batch_lines,
                )

                if len(conversation) < batch_lines:
                    raise ValueError(
                        "Incomplete result received. Batch lines: "
                        f"{len(conversation)}, expected: {batch_lines}"
                    )

                for i, translated_text in enumerate(conversation):
                    if i+1 > batch_lines:
                        break
                    translated_lines.append(list(translated_text.values())[0])

                progress_bar.update(batch_lines)

            except Exception as error:
                logger.error(str(error))

                first_start = segments_copy[max(0, i-(batch_lines-1))]["start"]
                logger.warning(
                    f"The batch from {first_start} to {last_start} "
                    "failed, is being corrected with Google Translate"
                )

                translator = GoogleTranslator(
                    source=fixed_source,
                    target=fixed_target
                )

                for txt_source in batch_conversation["conversation"]:
                    translated_txt = translator.translate(
                        list(txt_source.values())[0].strip()
                    )
                    translated_lines.append(translated_txt.strip())
                    progress_bar.update(1)

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines, fixed_target, fixed_source
    )


def gemini_sequential(segments, model, target, source=None):
    """
    Translate text segments sequentially using Google's Gemini API.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for segment text.
    - model (str): The Gemini model to use (e.g., "gemini-1.5-pro").
    - target (str): Target language code.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        logger.error(
            "Gemini requires the google-generativeai package. "
            "Install it with: pip install google-generativeai"
        )
        return translate_iterative(segments, target, source)

    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        return translate_iterative(segments, target, source)

    # Configure the Gemini API
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    segments_copy = copy.deepcopy(segments)
    target_name = INVERTED_LANGUAGES.get(target, target)
    source_name = INVERTED_LANGUAGES.get(source, source if source else "auto")

    for line_num, segment in enumerate(tqdm(segments_copy, desc="Translating with Gemini")):
        text = segment["text"].strip()
        
        system_prompt = (
            f"You are an expert translator. Translate the following text from "
            f"{source_name} to {target_name}. Maintain the tone, style, "
            f"and meaning of the original text. Only provide the translation, "
            f"no explanations or additional text."
        )
        
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                [
                    system_prompt,
                    text
                ]
            )
            
            translated_text = response.text.strip()
            segments_copy[line_num]["text"] = translated_text
            logger.debug(f"{text} >> {translated_text}")
            
        except Exception as error:
            logger.error(f"Error with Gemini API: {str(error)}")
            logger.warning("Falling back to Google Translate for this segment")
            translator = GoogleTranslator(source=source if source else "auto", target=target)
            segments_copy[line_num]["text"] = translator.translate(text)
    
    return segments_copy


def gemini_batch(segments, model, target, token_batch_limit=1000, source=None):
    """
    Translate a batch of text segments using Google's Gemini API.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for segment text.
    - model (str): The Gemini model to use (e.g., "gemini-1.5-pro").
    - target (str): Target language code.
    - token_batch_limit (int, optional): Maximum number of tokens for each batch.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.
    """
    try:
        import google.generativeai as genai
        import json
    except ImportError:
        logger.error(
            "Gemini requires the google-generativeai package. "
            "Install it with: pip install google-generativeai"
        )
        return translate_iterative(segments, target, source)

    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        return translate_iterative(segments, target, source)

    # Configure the Gemini API
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    segments_copy = copy.deepcopy(segments)
    target_name = INVERTED_LANGUAGES.get(target, target)
    source_name = INVERTED_LANGUAGES.get(source, source if source else "auto")
    
    # Batch segments
    batches = []
    current_batch = []
    current_token_count = 0
    
    for segment in segments:
        text = segment["text"].strip()
        # Rough token estimation (each word is ~1.3 tokens)
        estimated_tokens = len(text.split()) * 1.3
        
        if current_token_count + estimated_tokens > token_batch_limit and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_token_count = 0
        
        current_batch.append(text)
        current_token_count += estimated_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    # Process each batch
    all_translations = []
    for batch_idx, batch in enumerate(tqdm(batches, desc="Translating batches with Gemini")):
        batch_json = json.dumps({"texts": batch})
        
        system_prompt = (
            f"You are an expert translator. Translate the following texts from "
            f"{source_name} to {target_name}. Maintain the tone, style, "
            f"and meaning of the original texts. "
            f"Return ONLY a JSON object with the key 'translations' containing "
            f"an array of translated strings in the same order as the input."
        )
        
        prompt = (
            f"Translate the following texts to {target_name}:\n"
            f"{batch_json}\n\n"
            f"Return only a JSON object with the key 'translations' containing the translated texts."
        )
        
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content([system_prompt, prompt])
            
            response_text = response.text
            # Extract JSON from response
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                else:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0)
                
                translations_data = json.loads(response_text)
                batch_translations = translations_data.get("translations", [])
                
                if len(batch_translations) != len(batch):
                    logger.warning(
                        f"Batch {batch_idx}: Expected {len(batch)} translations, "
                        f"got {len(batch_translations)}. "
                        f"Using sequential translation for this batch."
                    )
                    # Fall back to sequential translation for this batch
                    batch_segments = [{"text": text} for text in batch]
                    translated_batch = gemini_sequential(batch_segments, model, target, source)
                    batch_translations = [segment["text"] for segment in translated_batch]
                
                all_translations.extend(batch_translations)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing Gemini response: {str(e)}")
                logger.warning(f"Response was: {response_text}")
                logger.warning("Using sequential translation for this batch")
                batch_segments = [{"text": text} for text in batch]
                translated_batch = gemini_sequential(batch_segments, model, target, source)
                all_translations.extend([segment["text"] for segment in translated_batch])
                
        except Exception as error:
            logger.error(f"Error with Gemini API batch {batch_idx}: {str(error)}")
            logger.warning("Falling back to sequential translation for this batch")
            batch_segments = [{"text": text} for text in batch]
            translated_batch = gemini_sequential(batch_segments, model, target, source)
            all_translations.extend([segment["text"] for segment in translated_batch])
    
    # Update segments with translations
    if len(all_translations) != len(segments_copy):
        logger.warning(
            f"Translation count mismatch: {len(all_translations)} translations for "
            f"{len(segments_copy)} segments. Falling back to sequential translation."
        )
        return gemini_sequential(segments, model, target, source)
    
    for i, translation in enumerate(all_translations):
        segments_copy[i]["text"] = translation
    
    return segments_copy


def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    """
    Translate text based on the specified translation process.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for segment
        text.
    - target (str): Target language code.
    - translation_process (str, optional): Translation method to use,
        "google_translator_batch" or "google_translator", default: "google_translator_batch"
    - chunk_size (int, optional): Maximum character limit for each translation
        chunk (default is 4500).
    - source (str, optional): Source language code. Defaults to None.
    - token_batch_limit (int, optional): Max token limit for the GPT models. Defaults to 1000.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Translates text segments using the specified translation process.
    - Uses either Google Translate (default) or GPT-based models for translation.
    """

    # import here, to be optional
    try:
        import os
        import openai
    except ImportError:
        logger.warning("openai is not installed, gpt options are not available.")
        logger.warning("Translation process switched to google_translator_batch.")
        if translation_process == "google_translator":
            return translate_iterative(segments, target, source)
        return translate_batch(segments, target, source=source)

    if target == source or translation_process == "disable_translation":
        return segments
    elif translation_process == "google_translator":
        return translate_iterative(segments, target, source)
    elif translation_process == "google_translator_batch":
        return translate_batch(segments, target, chunk_size, source)
    elif "gpt" in translation_process:
        client = openai.OpenAI()
        model_name = translation_process.replace("_batch", "")
        if "_batch" in translation_process:
            return gpt_batch(segments, model_name, target, token_batch_limit, source)
        else:
            return gpt_sequential(segments, model_name, target, source)
    elif "gemini" in translation_process:
        model_name = translation_process.replace("_batch", "")
        if "_batch" in translation_process:
            return gemini_batch(segments, model_name, target, token_batch_limit, source)
        else:
            return gemini_sequential(segments, model_name, target, source)
    else:
        logger.error(f"Unknown translation process: {translation_process}")
        return translate_batch(segments, target, source=source)
