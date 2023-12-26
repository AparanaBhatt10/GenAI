import openai
import ast
import re
import pandas as pd
import json

movie_df = pd.read_csv('filtered_movie_specs.csv')

"""
Thank you for providing that information. Based on your preferences, I have some movie recommendations for you. Here is the updated profile:

{
  'Rating': 'high',
  'Genre': ['Action'],
  'Duration': 'medium',
  'Language': ['English']
}

With this information, I can suggest some high-rated, light-hearted action movies with a medium duration that are in English.


"""
def  initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"
    example_user_req = {'Rating': 'average', 'Genre':['Adventure', 'Comedy', 'Crime', 'Family'], 'Duration': 'long', 'Language': ['English'] }

    system_message = f"""

    You are an intelligent disney plus movie recommendation assistant and your goal is to recommend the movie to the user based on input preferences.
    You need to ask relevant questions and understand the user need by analysing the user's responses.
    You final objective is to fill the values for the different keys ('Rating','Genre','Duration','Language') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'Rating': 'values','Genre': 'values','Duration': 'values','Language': 'values'}}
    The value of 'Rating' should be one of 'poor','average','high'.The value of 'Duration' should be one of 'long','medium','short'.
    The value of 'Language' should be a list of strings representing the language names as input by the user,
    The value of 'Genre' should be a list of strings representing the genre of movie.
    The values currently in the dictionary are only representative values. 
    
    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
     - The value of 'Rating' should be one of 'poor','average','high'.The value of 'Duration' should be one of 'long','medium','short'.
     - The value of 'Language' should be a list of strings representing the language names as input by the user,
    
     - The value of 'Genre' should be a list of strings representing the genre of movie. 
     - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary motive of movie recommendation is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys ('Rating','Genre','Duration','Language','Plot') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys. 
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step. 
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same. 
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary. 
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I want to see a movie can you recommend me some movie?"
    Assistant: "That's great, watching good movie elevates a person's mood, I can try to suggest you some good movies.Can you explain me more about what you want to watch?"
    User: "I want to see a movie that is not much long , and is light hearted"
    Assistant: "Thank you for providing that information. That's great choice , do you prefer comdey movies of romantic movies?"
    User: "Umm... I am not sure can you suggest some movies of both genres and I can pick one myself."
    Assistant: "Got you ! That's great. Alebit, could you tell me about your language preferences as well, it will help me cater to your request better?"
    User: "Well yeah sure, I prefer English movies and also I want to see some high  rated movie well."
    Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements. Do not start with Assistant: "
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation


def get_chat_model_completions(messages):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
        max_tokens=300
    )

    return response.choices[0].message.content


def moderation_check(user_input):
    response = openai.moderations.create(input=user_input)
    moderation_output = response.results[0]
    if moderation_output.flagged:
        return "Flagged"
    else:
        return "Not Flagged"


def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'Rating','Duration','Language','Genre'
    Next you need to evaluate if the keys have the the values filled correctly.
    The value of 'Rating' should be one of 'poor','average','high'.The value of 'Duration' should be one of 'long','medium','short'.
    The value of 'Language' should be a list of strings representing the language names as input by the user,
    The value of 'Genre' should be a list of strings representing the genre of movie. 
    Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
   
    Output a string 'Yes' if the input contains the dictionary with the values correctly filled for all keys.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """
    list = [prompt]

    confirmation = openai.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0
    )

    return confirmation.choices[0].text


def dictionary_present(response):
    delimiter = "####"
    user_req = {'Rating': 'Average', 'Genre':['Documentary', ' Music'], 'Duration': 'Short', 'Language':['English']}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values 
            following same format, in the keys having list values do not need to strictly check the value in the list.
        

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - Rating: High - Genre: ['Music'] - Duration: Short - Language: ['English']
            output: {{'Rating': 'high', 'Genre': ['Music'], 'Duration': 'Short', 'Language': 'English'}}

            input: {{'Rating':     'poor', 'Genre':     'Music', 'Duration':    'Short', 'Language': 'English'}}
            output: {{'Rating': 'poor', 'Genre': ['Music'], 'Duration': 'Short', 'Language': ['English']}}

            input: Here is your user profile 'Rating': 'high','Genre': 'Music','Duration': 'Short','Language': 'English'
            output: {{'Rating': 'High','Genre': ['Music'],'Duration':'Short','Language': ['English']}}
            
            input: Here is the updated profile:{'Rating': 'high','Genre': ['Action'],'Duration': 'medium','Language': ['English']} With this information I can suggest you movies
            output: {{'Rating': 'high','Genre': ['Action'],'Duration': 'medium','Language': ['English']}}
            
            {delimiter}

            Here is the input {response}

            """

    list = [prompt]
    response = openai.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0
    )
    return response.choices[0].text


def extract_dictionary_from_string(string):
    print
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary






def compare_movies_with_user_req(user_req_string):

    user_requirements = extract_dictionary_from_string(user_req_string)
    filtered_movie = movie_df.copy()

    # These lines create a copy of the movie_df already created at the top, it will prveent us reading csv repeatedly
    # We can then check for the dictionary values

    mappings = {
        'poor': 0,
        'average': 1,
        'high': 2,
        'long':2,
        'short':0,
        'medium':1,
    }
    # Create 'Score' column in the DataFrame and initialize to 0
    filtered_movie['Score'] = 0
    for index, row in filtered_movie.iterrows():
        user_product_match_str = row['movie_specs']
        movie_values = extract_dictionary_from_string(user_product_match_str)
        score = 0

        for key, user_value in user_requirements.items():

            laptop_value = movie_values.get(key, None)
            laptop_mapping = mappings.get(laptop_value.lower(), -1)
            user_mapping = mappings.get(user_value.lower(), -1)
            if laptop_mapping >= user_mapping:
                ### If the laptop value is greater than or equal to the user value the score is incremented by 1
                score += 1

        filtered_movie.loc[index, 'Score'] = score

    # Sort the laptops by score in descending order and return the top 3 products
    top_movies = filtered_movie.drop('movie_specs', axis=1)
    top_movies = top_movies.sort_values('Score', ascending=False).head(3)

    return top_movies.to_json(orient='records')

def recommendation_validation(laptop_recommendation):
    data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1


def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent disney plus movie recoomendation bot and are tasked to \
    recommend movies to the user  from the catalogue: {products}.\
    You should keep the user profile in mind while answering the questions.\

    Start with the brief plot of the movie and follow the below format to show the recommendations:\:
    1. <Movie Name> : <Plot>, <Rating>, <Genre>, <Director>, <Runtime>
    2. <Movie Name> : <Plot>, <Rating>, <Genre>, <Director>, <Runtime>

    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation
