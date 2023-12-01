from statistics import mode
import unidecode
import string


def generate_ngrams(text, WordsToCombine):
    """
    Returns a list of n-grams of length WordsToCombine generated from the input text.
    Each n-gram is a list of WordsToCombine consecutive words from the input text.

    Parameters:
    -----------
        text (str): the input text to be split into n-grams.
        WordsToCombine (int): the number of words to be combined in each n-gram.

    Returns:
    ----------
        list: a list of n-grams generated from the input text.
    """
    words = text.split()
    output = []
    for i in range(len(words) - WordsToCombine+1):
        output.append(words[i:i+WordsToCombine])
    return output


def make_keys(text, WordsToCombine):
    """
    Returns a list of keys generated from n-grams of length WordsToCombine in the input text.
    Each key is a string of WordsToCombine consecutive words from the input text.

    Parameters:
    -----------
        text (str): the input text to generate keys from.
        WordsToCombine (int): the number of words to combine in each key.

    Returns:
    ----------
        list: a list of keys generated from the input text.
    """
    gram = generate_ngrams(text, WordsToCombine)
    sentences = []
    for i in range(len(gram)):
        sentence = ' '.join(gram[i])
        sentences.append(sentence)
    return sentences


def rule_based_prediction(input, keys):
    """
    A rule-based prediction function that takes an input 
    string and a dictionary of keys and values.

    Arguments:
    ----------

        input: a string of words.
        keys: a dictionary of keys and values.

    Returns:
    ----------
        int: the predicted value for the input.

    Explanation:
    ----------
    0. Step

    - Create two empty lists, `values` and `sentences`.
    - Preprocess the input by removing punctuation and converting it to lowercase,
        since this is the format of the keys in the `keys` dictionary.

    1. Step

    - If the input is a single word, check if it is in the keys dictionary.
    - If it is, then it is a valid key, and we can use it to append 
        100 copies of the value associated with that key to the `values` list.
    - Finally, we add the input to itself, for further processing.
    - If the input is not a single word, then we check if it is in  the keys 
        dictionary. If it is, we can use it to append 100 copies of 
        the value associated with that key to the `values` list.
    - This means that if the input is already a valid key, we already know  
        what the output should be.

    2. Step
    - We will us the `make_keys` function to generate keys from the input.
    - The `make_keys` function will generate keys of length 1 to the length
        of the input + 1.
    - We will append the generated keys to the `sentences` list.

    3. Step

    - We will iterate over the `sentences` list, and for each key in the list,
        we will check if it is in the keys dictionary.
    - If it is, we can use it to append the value associated with that key to
        the `values` list.

    4. Step

    - We flaten the `values` list, and if it is empty, we know that no match
        was found in our dictionary. Therfore, is not "domain question".
    - If it is not empty, we know that a match was found in our dictionary.
        Therfore, is a "domain question".
    - If the input is a "domain question", take the mode (most common) value
        in the `values` list, and use it to generate the output. These value
        keys are associeted with pre-defined answers.

    5. Step

    - Return 0 or the index of the awnser to be used.

    """
    sentences = list()
    values = list()

    x = input.translate(str.maketrans('', '', string.punctuation))
    x = x.lower()
    x = unidecode.unidecode(x)

    if len(x.split()) == 1:
        if x in keys.keys():
            values.append([keys[x]] * 100)
        x = x + ' ' + x

    else:
        if x in keys.keys():
            values.append([keys[x]] * 100)

    for i in range(1, len(x.split()) + 1):
        sentences.append(make_keys(x, i))

    for sentence in sentences:
        for key in sentence:
            if key in keys.keys():
                values.append([keys[key]])

    if len([item for sublist in values for item in sublist]) == 0:
        # print(f""" No match found for input: '{input}'.""")
        return 0

    else:
        # print(f"""Match found for input: '{input}'.""")
        return int(mode([item for sublist in values for item in sublist])) - 1


def rule_based_prediction_app(input, keys, answers):
    """
    A rule-based prediction function that takes an input 
    string and a dictionary of keys and values.

    Arguments:
    ----------

        input: a string of words.
        keys: a dictionary of keys and values.
        answers: a list of answers to be used.

    Returns:
    ----------
        str: the predicted answer for the input.

    Explanation:
    ----------
    0. Step

    - Create two empty lists, `values` and `sentences`.
    - Preprocess the input by removing punctuation and converting it to lowercase,
        since this is the format of the keys in the `keys` dictionary.

    1. Step

    - If the input is a single word, check if it is in the keys dictionary.
    - If it is, then it is a valid key, and we can use it to append 
        100 copies of the value associated with that key to the `values` list.
    - Finally, we add the input to itself, for further processing.
    - If the input is not a single word, then we check if it is in  the keys 
        dictionary. If it is, we can use it to append 100 copies of 
        the value associated with that key to the `values` list.
    - This means that if the input is already a valid key, we already know  
        what the output should be.

    2. Step
    - We will us the `make_keys` function to generate keys from the input.
    - The `make_keys` function will generate keys of length 1 to the length
        of the input + 1.
    - We will append the generated keys to the `sentences` list.

    3. Step

    - We will iterate over the `sentences` list, and for each key in the list,
        we will check if it is in the keys dictionary.
    - If it is, we can use it to append the value associated with that key to
        the `values` list.

    4. Step

    - We flaten the `values` list, and if it is empty, we know that no match
        was found in our dictionary. Therfore, is not "domain question".
    - If it is not empty, we know that a match was found in our dictionary.
        Therfore, is a "domain question".
    - If the input is a "domain question", take the mode (most common) value
        in the `values` list, and use it to generate the output. These value
        keys are associeted with pre-defined answers.

    5. Step

    - Return an "IDK" answers or the index of the answers to be used.
    - 

    """
    sentences = list()
    values = list()

    x = input.translate(str.maketrans('', '', string.punctuation))
    x = x.lower()
    x = unidecode.unidecode(x)

    if len(x.split()) == 1:
        if x in keys.keys():
            values.append([keys[x]] * 100)
        x = x + ' ' + x

    else:
        if x in keys.keys():
            values.append([keys[x]] * 100)

    for i in range(1, len(x.split()) + 1):
        sentences.append(make_keys(x, i))

    for sentence in sentences:
        for key in sentence:
            if key in keys.keys():
                values.append([keys[key]])

    if len([item for sublist in values for item in sublist]) == 0:
        # print(f""" No match found for input: '{input}'.""")
        return 'Sorry, I did not understand your question. Please, try again.'

    else:
        # print(f"""Match found for input: '{input}'.""")
        return answers[int(mode([item for sublist in values for item in sublist])) - 1]


def standerdize_text(text):
    """
    This function takes in a string of text and performs the following cleaning steps:
    1. Removes all punctuation marks from the text
    2. Converts the text to lowercase
    3. Removes any accents or diacritics from the text

    Parameters:
    -----------
    text : str
        The string of text to be cleaned

    Returns:
    --------
    text : str
        The standerdize version of the input text
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return unidecode.unidecode(text)


def toggle_modal(n1, n2, is_open):
    """
    Toggles the visibility of the simple modal window.

    Args:
    ---------
        n1 (bool): A flag to check if the first modal is open or closed.
        n2 (bool): A flag to check if the second modal is open or closed.
        is_open (bool): A flag to check if the modal is currently open or not.

    Returns:
    ---------
        bool: The updated value of the `is_open` flag after toggling the modal window.
    """
    if n1 or n2:
        return not is_open
    return is_open
