from fastapi import FastAPI
from api.schemas import *
from discovery.products import extract_products
from discovery.personas import generate_personas
from discovery.topics import generate_topics
from discovery.company import verify_company_from_url
from analysis.analyze import run_analysis
from analysis.report import generate_report
from analysis.prompts import generate_prompts
from llm.llm_factory import get_llm

app = FastAPI(title="GEO Intelligence Core")

class CompanyVerifyRequest(BaseModel):
    url: str

@app.get("/")
def greeting():
    return "Welcome to GEO Intelligence Application"

@app.post("/verify-company")
def verify_company(req: CompanyVerifyRequest):
    return verify_company_from_url(req.url)

@app.post("/products")
def products(req: ProductRequest):
    return {"products": extract_products(req.company)}

@app.post("/personas")
def personas(req: PersonaRequest):
    return {"personas": generate_personas(req.company, req.product)}

@app.post("/topics")
def topics(req: TopicRequest):
    return {"topics": generate_topics(req.company, req.product, req.persona)}

#  FIXED: PROMPTS SHOULD RETURN ONLY PROMPTS 
@app.post("/prompts")
def prompts(req: AnalysisRequest):
    results = []
    total = 0

    for model in req.models:
        llm = get_llm(model)

        prompts = generate_prompts(
            product=req.product,
            persona=req.persona,
            topic=req.topic,
            num=req.num_prompts,
            llm=llm,
        )

        results.append({
            "model": model,
            "prompts": prompts
        })

        total += len(prompts)

    return {
        "total_prompts": total,
        "results": results
    }

# REPORT: It now receives personas explicitly

@app.post("/report")
def report(payload: ReportRequest):
    return generate_report(payload.dict())