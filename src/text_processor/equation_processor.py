#!/usr/bin/python3
import re


def generalize(equation, input_numbers):
    '''

    :param equation: the equation to generalize .
    :param input_numbers:  the numbers occuring in the question in serial order.
    :return:
    '''

    # remove whitespaces altogether
    equation = equation.replace(' ', '')

    # replace all the variable names with X
    format_words = re.sub(r'[a-zA-Z_]+[a-zA-Z0-9_]*', 'X', equation)

    # split the equation from using =
    v1, v2 = format_words.split('=')

    # if a side of equation starts with bracket and ends with bracket, remove the bracket.
    if v1[0] == '(' and v1[-1] == ')':
        v1 = v1[1:-1]
    if v2[0] == '(' and v2[-1] == ')':
        v2 = v2[1:-1]

    # combine the splitted parts
    ans = v1 + '=' + v2

    # the code below  could make you psyco. so avoid it.
    tokens = []
    last = 0
    for index in re.finditer(r'[0-9.]+', ans):
        tokens.append(ans[last:index.start(0)])
        tokens.append(ans[index.start(0):index.end(0)])
        last = index.end(0)
    tokens.append(ans[last:])
    tokens = [x for x in tokens if x != '']
    numbers = [x for x in range(len(tokens)) if re.match('[0-9.]+', tokens[x]) is not None]

    # make sure that the number of digits in equation equals that in the question.
    if len(numbers) is not len(input_numbers):
        print("Mapping fault on :",numbers,input_numbers,equation)
        raise ValueError("All digit parameters not used", "Parameter Mapping Fault")
    sorted(numbers)

    # now map the numbers in the sentence to the ones in the equation by index.
    for number_index in numbers:
        try:
            index = input_numbers.index(float(tokens[number_index]))
        except ValueError as e:
            raise ValueError("Cannot map input number to numbers in equaiton",
                            "Input ->" + str(equation) + ',' + str(input_numbers), e)

        tokens[number_index] = 'N' + str(index)

    ans = ''
    for a in tokens:
        ans += a
    return ans


if __name__ == "__main__":
    equation = 'X=(9.0+29.0)'
    print(equation, '-->', generalize(equation, [9.0, 29.0]))

    equation = 'cost_of_1_lollipop=90.0/120.0'
    print(equation, '-->', generalize(equation, [120.0, 90.0]))

    equation = 'cost_of_1_lollipop=90.0/120.0+A_BC_D+30.2+1/2'
    print(equation, '-->', generalize(equation, [120.0, 90.0, 1, 2, 30.2]))
