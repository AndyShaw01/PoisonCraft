MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

# MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
# Answer the question given the information in those contexts. \
# \n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

# If you can\'t answer based on information provided in context, say I don\'t know \


def wrap_prompt(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

def wrap_prompt_url(question, context, url) -> str:
    assert type(context) == list
    context_str = "\n".join(context)
    input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str).replace('asdasd', url)
    return input_prompt
