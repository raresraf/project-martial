#include <iostream>
#include <string>

using namespace std;

int main() {
  string name;
  cin >> name;
  int length = name.length();
  for (int i = 0; i < length; i++) {
    char temp = name.at(i);
    for (int j = i + 1; j < length; j++) {
      char holder = name.at(j);
      if (holder == temp) {
        name.erase(j, 1);
        j--;
        length--;
      }
    }
  }
  if (name.length() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
