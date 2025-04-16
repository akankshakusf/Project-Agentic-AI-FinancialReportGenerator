#import packages
import os 
from dotenv import load_dotenv
import autogen
from autogen import ConversableAgent,AssistantAgent
from datetime import datetime
import streamlit as st 
import zipfile 

# import environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# llm configuration
llm_config = {
    "model":"gpt-4o-mini",
    "api_key":api_key
}

writing_tasks = [
        """Develop an engaging financial report using all information provided, include the normalized_prices.png figure,
        and other figures if provided.
        Mainly rely on the information provided. 
        Create a table comparing all the fundamental ratios and data.
        Provide comments and description of all the fundamental ratios and data.
        Compare the stocks, consider their correlation and risks, provide a comparative analysis of the stocks.
        Provide a summary of the recent news about each stock. 
        Ensure that you comment and summarize the news headlines for each stock, provide a comprehensive analysis of the news.
        Provide connections between the news headlines provided and the fundamental ratios.
        Provide an analysis of possible future scenarios. 
        """]

## Agent 1
financial_assistant = autogen.AssistantAgent(
    name="Financial_assistant",
    llm_config=llm_config,
)

## Agent 2
research_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
)

## Agent 3
writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer, known for
        your insightful and engaging finance reports.
        You transform complex concepts into compelling narratives. 
        Include all metrics provided to you as context in your analysis.
        Only answer with the financial report written in markdown directly, do not include a markdown language block indicator.
        Only return your final work without additional comments.
        """,
)

## Agent 4
export_assistant = autogen.AssistantAgent(
    name="Exporter",
    llm_config=llm_config,
)

## Agent 5
critic = autogen.AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content.",
)

## Agent 6
legal_reviewer = autogen.AssistantAgent(
    name="Legal_Reviewer",
    llm_config=llm_config,
    system_message="You are a legal reviewer, known for "
        "your ability to ensure that content is legally compliant "
        "and free from any potential legal issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

## Agent 7
consistency_reviewer = autogen.AssistantAgent(
    name="Consistency_reviewer",
    llm_config=llm_config,
    system_message="You are a consistency reviewer, known for "
        "your ability to ensure that the written content is consistent throughout the report. "
        "Refer numbers and data in the report to determine which version should be chosen " 
        "in case of contradictions. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

## Agent 8
textalignment_reviewer = autogen.AssistantAgent(
    name="Text_lignment_reviewer",
    llm_config=llm_config,
    system_message="You are a text data alignment reviewer, known for "
        "your ability to ensure that the meaning of the written content is aligned "
        "with the numbers written in the text. " 
        "You must ensure that the text clearely describes the numbers provided in the text "
        "without contradictions. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

## Agent 9
completion_reviewer = autogen.AssistantAgent(
    name="Completion_Reviewer",
    llm_config=llm_config,
    system_message="You are a content completion reviewer, known for "
        "your ability to check that financial reports contain all the required elements. "
        "You always verify that the report contains: a news report about each asset, " 
        "a description of the different ratios and prices, "
        "a description of possible future scenarios, a table comparing fundamental ratios and "
        " at least a single figure. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

## Agent 10
meta_reviewer = autogen.AssistantAgent(
    name="Meta_Reviewer",
    llm_config=llm_config,
    system_message="You are a meta reviewer, you aggregate and review "
    "the work of other reviewers and give a final suggestion on the content.",
)

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

#define the chat structure
review_chats = [
    {
    "recipient": legal_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'Reviewer': '', 'Review': ''}.",},
     "max_turns": 1},
    {"recipient": textalignment_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'reviewer': '', 'review': ''}",},
     "max_turns": 1},
    {"recipient": consistency_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'reviewer': '', 'review': ''}",},
     "max_turns": 1},
    {"recipient": completion_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'reviewer': '', 'review': ''}",},
     "max_turns": 1},
     {"recipient": meta_reviewer, 
      "message": "Aggregrate feedback from all reviewers and give final suggestions on the writing.", 
     "max_turns": 1},
]


# Register nested chat sessions to critic agent
critic.register_nested_chats(
    review_chats,   # List of nested chats to review
    trigger=writer, # Agent that triggers the reviews
)

# ===

user_proxy_auto = autogen.UserProxyAgent(
    name="User_Proxy_Auto",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },  
)



#-------------------# Streamlit UI #-------------------#

st.set_page_config("Bedrock GenBlog")

## Display the image and title
col1, col2 = st.columns([1, 6])
with col1:
    st.image("openai.png", width=100)  # Adjust the width as needed
with col2:
    st.title("FinSight Pro: Multi-Agent Financial Report Generator")#displayed on app
##-----------------------

assets= st.text_input("Assets you want to analyze(provide the tickers)?")
hit_button = st.button("Start Analysis")

if hit_button is True:
    date_str = datetime.now().strftime("%Y-%m-%d")

    financial_tasks = [
        f"""Today is the {date_str}. 
        What are the current stock prices of {assets}, and how is the performance over the past 6 months in terms of percentage change? 
        Start by retrieving the full name of each stock and use it for all future requests.
        Prepare a figure of the normalized price of these stocks and save it to a file named normalized_prices.png. Include information about, if applicable: 
        * P/E ratio
        * Forward P/E
        * Dividends
        * Price to book
        * Debt/Eq
        * ROE
        * Analyze the correlation between the stocks
        Do not use a solution that requires an API key.
        If some of the data does not makes sense, such as a price of 0, change the query and re-try.""",

        """Investigate possible reasons of the stock performance leveraging market news headlines from Bing News or Google Search. Retrieve news headlines using python and return them. Use the full name stocks to retrieve headlines. Retrieve at least 10 headlines per stock. Do not use a solution that requires an API key. Do not perform a sentiment analysis.""",
    ]

    with st.spinner("Agents working on the analysis...."):
        chat_results = autogen.initiate_chats([
            {
                "sender": user_proxy_auto,
                "recipient": financial_assistant,
                "message": financial_tasks[0],
                "silent": False,
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "summary_prompt": "Return the stock prices of the stocks, their performance and all other metrics "
                                      "into a JSON object only. Provide the name of all figure files created. Provide the full name of each stock.",
                },
                "clear_history": False,
                "carryover": "Wait for confirmation of code execution before terminating the conversation. Verify that the data is not completely composed of NaN values. Reply TERMINATE in the end when everything is done."
            },
            {
                "sender": user_proxy_auto,
                "recipient": research_assistant,
                "message": financial_tasks[1],
                "silent": False,
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "summary_prompt": "Provide the news headlines as a paragraph for each stock, be precise but do not consider news events that are vague, return the result as a JSON object only.",
                },
                "clear_history": False,
                "carryover": "Wait for confirmation of code execution before terminating the conversation. Reply TERMINATE in the end when everything is done."
            },
            {
                "sender": critic,
                "recipient": writer,
                "message": writing_tasks[0],
                "carryover": "I want to include a figure and a table of the provided data in the financial report.",
                "max_turns": 2,
                "summary_method": "last_msg",
            }
        ])

        # === Safe extraction of final report ===
        try:
            if chat_results and chat_results[-1].chat_history:
                final_report = chat_results[-1].chat_history[-1]["content"]

                # Display in Streamlit UI
                st.markdown(final_report)

                # Save markdown + embed image ref
                md_path = "./coding/final_report.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(final_report + "\n\n![Normalized Prices]")

                st.success("Markdown saved as final_report.md")

                image_path = "./coding/normalized_prices.png"
                st.image(image_path, caption="Normalized Prices")

                # Zip markdown + image
                zip_path = "./coding/financial_report_bundle.zip"
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    zipf.write(md_path, arcname="final_report.md")
                    zipf.write(image_path, arcname="normalized_prices.png")

                # Download button for the zip file
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download Report (Markdown + PNG)",
                        data=f,
                        file_name="financial_report_bundle.zip",
                        mime="application/zip"
                    )
            else:
                st.error("No content generated. Try again or check agent configuration.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.warning("Please enter at least one asset ticker before starting analysis.")