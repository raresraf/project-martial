#include <iostream>
#include <string>

int main() {
  std::string input;

  unsigned counter1 = 0;
  unsigned counter2 = 0;
  unsigned counter3 = 0;

  std::cin >> input;

  for (int i = 0; i < input.length(); i += 2) {
    if ('1' == input[i]) {
      counter1++;
    } else if ('2' == input[i]) {
      counter2++;
    } else if ('3' == input[i]) {
      counter3++;
    }
  }

  for (int i = 0; i < counter1; i++) {
    std::cout << (i ? "+1" : "1");
  }

  for (int i = 0; i < counter2; i++) {
    std::cout << (counter1 || i ? "+2" : "2");
  }

  for (int i = 0; i < counter3; i++) {
    std::cout << (counter1 || counter2 || i ? "+3" : "3");
  }

  return 0;
}
