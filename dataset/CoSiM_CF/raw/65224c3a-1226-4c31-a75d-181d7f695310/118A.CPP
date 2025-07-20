#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  getline(cin, str);

  for (int i = 0; i < str.length(); i++) {
    if (str[i] == 'a' || str[i] == 'A' || str[i] == 'o' || str[i] == 'O' ||
        str[i] == 'u' || str[i] == 'U' || str[i] == 'y' || str[i] == 'Y' ||
        str[i] == 'e' || str[i] == 'E' || str[i] == 'i' || str[i] == 'I') {
      str.erase(i, 1);
      i--;
    } else {
      if (str[i] >= 'A' && str[i] <= 'Z') {
        str[i] -= ('A' - 'a');
      }

      str.insert(i, ".");
      i++;
    }
  }

  cout << str;

  return 0;
}