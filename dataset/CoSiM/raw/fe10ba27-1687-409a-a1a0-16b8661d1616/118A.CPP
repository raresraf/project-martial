#include <cstring>
#include <iostream>

using namespace std;

int main() {
  string input;
  cin >> input;
  for (int i = 0; i < input.size(); i++) {
    char ch = tolower(input[i]);
    if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
        ch == 'y') {
      continue;
    } else {
      cout << "." << ch;
    }
  }
  return 0;
}
