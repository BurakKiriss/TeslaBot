from langchain_core.prompts import PromptTemplate


CONFIDENCE_GATE_THRESHOLD = 55.0


INSUFFICIENT_INFO_TOKEN = "INSUFFICIENT_INFO"

NO_ANSWER_MESSAGE = (
    "Bu konuda kılavuzlarda yeterli veya net bir bilgi bulamadım. "
    "Sorunuzu farklı bir şekilde ifade etmeyi veya daha spesifik bir "
    "araç modeli/özellik belirtmeyi deneyebilirsiniz."
)



EXAMPLE_ANSWER_TEXTS = [
    "In Performance trim, Model Y accelerates from 0-60 mph in 3.5 seconds.",
]

ANSWER_PROMPT = PromptTemplate.from_template(
    "You are a Tesla owner's manual assistant. You must answer STRICTLY and "
    "ONLY using the information contained in the CONTEXT below. The context "
    "consists of real excerpts from official Tesla owner's manuals.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the information in the CONTEXT. Do not use any outside knowledge.\n"
    "- Do not guess, infer, or add any information that is not explicitly stated in the CONTEXT.\n"
    "- The CONTEXT may contain numbers, facts, or details that are RELATED to the "
    "topic of the question but do NOT actually answer what was asked (for example, "
    "a distance limit for towing when the question asks about driving range). "
    "Do NOT include such unrelated numbers or facts in your answer just because "
    "they mention similar keywords. Only use information that directly and "
    "specifically answers the QUESTION.\n"
    "- If the CONTEXT does not contain information that directly and specifically "
    "answers the QUESTION, you MUST respond with exactly this token and nothing "
    "else: {insufficient_token}\n"
    "- The EXAMPLES below are ONLY to show you the response format. They are NOT "
    "real data. NEVER reuse any fact, number, or specific detail from the EXAMPLES "
    "in your actual answer, even if the topic seems similar to the real QUESTION. "
    "Only use facts that appear in the real CONTEXT section below.\n"
    "- Do not apologize, do not explain, do not add greetings or filler text.\n"
    "- Keep the answer concise and factual. Prefer short paragraphs or bullet points.\n"
    "- If relevant, mention which Tesla model the information applies to.\n\n"
    "EXAMPLE 1 (context has a directly relevant answer):\n"
    "CONTEXT: [Source: model_y.pdf, Page: 85] Model Y accelerates from 0-60 mph "
    "in 3.5 seconds in Performance trim.\n"
    "QUESTION: How fast does Model Y accelerate from 0-60 mph?\n"
    "ANSWER: In Performance trim, Model Y accelerates from 0-60 mph in 3.5 seconds.\n\n"
    "EXAMPLE 2 (context mentions similar keywords but does NOT answer the question "
    "- this is the case you must watch for):\n"
    "CONTEXT: [Source: model_x.pdf, Page: 241] If Model X must be transported "
    "without a flatbed truck, wheel lifts and dollies must be used. This method "
    "may only be used for a maximum of 35 miles (55 km).\n"
    "QUESTION: What is the maximum driving range of Model X?\n"
    "ANSWER: {insufficient_token}\n\n"
    "CRITICAL: Towing-related documents often mention SEVERAL different weight "
    "specifications that are easy to confuse: 'maximum towing capacity' (the "
    "trailer weight), 'maximum tongue weight' (downward force on the hitch), "
    "and 'accessory/carrier maximum weight' (for bike racks, ski carriers, "
    "etc., which is a completely different and usually much smaller number). "
    "NEVER use an accessory/carrier weight limit to answer a question about "
    "towing capacity, or vice versa. If the CONTEXT only gives you an "
    "accessory carrier weight limit but the QUESTION asks about towing "
    "capacity (or which vehicle can tow more), you do NOT have the answer.\n\n"
    "EXAMPLE 2B (context has an accessory carrier weight limit, which is NOT "
    "the same thing as towing capacity - do not confuse them):\n"
    "CONTEXT: [Source: model_3.pdf, Page: 99] The Model 3 tow hitch supports "
    "accessory carriers with a maximum vertical load of 60 kg.\n\n"
    "[Source: model_y.pdf, Page: 99] The Model Y tow hitch supports accessory "
    "carriers with a maximum vertical load of 72 kg.\n"
    "QUESTION: Which Tesla model can tow more weight?\n"
    "ANSWER: {insufficient_token}\n\n"
    "EXAMPLE 3 (context has SEVERAL sources from different models, but NONE of "
    "them actually compares the models against each other - you must NOT guess "
    "or pick a 'winner' just because a model name appears in the context):\n"
    "CONTEXT: [Source: model_3.pdf, Page: 82] To maximize range, slow down and "
    "avoid rapid acceleration.\n\n"
    "[Source: model_s.pdf, Page: 89] To maximize range, slow down and avoid "
    "rapid acceleration.\n\n"
    "[Source: model_y.pdf, Page: 85] To maximize range, slow down and avoid "
    "rapid acceleration.\n"
    "QUESTION: Which Tesla model has the longest range?\n"
    "ANSWER: {insufficient_token}\n\n"
    "IMPORTANT DISTINCTION: EXAMPLE 3 above shows a case where NONE of the "
    "sources contain a fact that answers the question for any model. This is "
    "DIFFERENT from a case where the CONTEXT contains explicit, model-specific "
    "numeric facts (such as a capacity, weight, speed, or range figure) for two "
    "or more different models, where each fact DOES directly answer the aspect "
    "asked about for its own model. In that case, you SHOULD combine those "
    "facts into a comparison, even if no single source explicitly compares them "
    "in one sentence. See EXAMPLE 4 below.\n\n"
    "EXAMPLE 4 (context has explicit numeric facts for two different models - "
    "combine them into a comparison):\n"
    "CONTEXT: [Source: model_x.pdf, Page: 119] The maximum towing capacity for "
    "Model X is 2300 kg.\n\n"
    "[Source: model_y.pdf, Page: 100] These values are for a 1600 kg maximum "
    "braked trailer.\n"
    "QUESTION: Compare the maximum towing capacity of Model X and Model Y.\n"
    "ANSWER: Model X has a higher maximum towing capacity (2300 kg) than "
    "Model Y (1600 kg).\n\n"
    "Now answer the real question below using only the real context.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)


def _build_context_block(ranked_results):
    """
    Reranker'dan gelen (chunk, raw_score, confidence_pct) üçlülerini,
    LLM'e verilecek tek bir CONTEXT metnine dönüştürür. Her parçanın
    kaynağı (dosya + sayfa) açıkça etiketlenir; bu hem modelin kaynağı
    karıştırmasını azaltır hem de ileride cevaba kaynak eklemek istersek
    zemin hazırlar.
    """
    blocks = []
    for chunk, _raw_score, _confidence_pct in ranked_results:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "unknown")
        blocks.append(f"[Source: {source}, Page: {page}]\n{chunk.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query, ranked_results, llm):
    """
    Reranker'dan gelen sıralı sonuçlara göre, sadece kılavuz içeriğine
    dayanan bir cevap üretir. Güven skoru yetersizse ya da model
    INSUFFICIENT_INFO sinyali dönerse, standart bir "bulunamadı" mesajı
    döndürülür (halüsinasyon üretmek yerine).

    Dönüş: (answer_text: str, was_answered: bool)
    """
    if not ranked_results:
        return NO_ANSWER_MESSAGE, False

    
    best_confidence = ranked_results[0][2]
    if best_confidence < CONFIDENCE_GATE_THRESHOLD:
        print(f"[Generator] Güven kapısı reddetti (en iyi skor: %{best_confidence} < %{CONFIDENCE_GATE_THRESHOLD}). LLM çağrılmadı.")
        return NO_ANSWER_MESSAGE, False

    
    context = _build_context_block(ranked_results)
    prompt = ANSWER_PROMPT.format(
        context=context,
        question=query,
        insufficient_token=INSUFFICIENT_INFO_TOKEN,
    )

    print("\n[Generator] Cevap üretiliyor...")
    raw_answer = llm.invoke(prompt).strip()
    print(f"--- Ham Model Çıktısı ---\n{raw_answer}\n-------------------------")

    
    if INSUFFICIENT_INFO_TOKEN in raw_answer:
        print("[Generator] Model INSUFFICIENT_INFO sinyali döndürdü.")
        return NO_ANSWER_MESSAGE, False

    
    normalized_answer = raw_answer.strip().rstrip(".").lower()
    for example_text in EXAMPLE_ANSWER_TEXTS:
        if normalized_answer == example_text.strip().rstrip(".").lower():
            print("[Generator] Model, prompt içindeki örnek cevabı birebir kopyaladı. Reddedildi.")
            return NO_ANSWER_MESSAGE, False

    return raw_answer, True