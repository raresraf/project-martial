#include <iostream>
#include <map>
#include <string>

int main() {
  std::string input;
  std::map<char, int> container;
  std::map<char, int>::iterator it;

  char currentChar;

  std::cin >> input;

  for (int i = 0; i < input.length(); i++) {

    currentChar = input[i];
    it = container.find(currentChar);

    if (it == container.end()) /* if element not found in map add it */
    {
      container[currentChar] = 1;
    }
    // else do nothing
  }

  std::cout << ((0 == container.size() % 2) ? "CHAT WITH HER!" : "IGNORE HIM!");

  return 0;
}
