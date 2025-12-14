#include <iostream>

using namespace std;

int main() {
  string str;
  cin >> str;
  if ((int)str.at(0) >= 97) {
    int ascii = str.at(0) - 32;
    char asciiToChar = static_cast<char>(ascii);
    str.at(0) = asciiToChar;
    cout << str << endl;
  } else
    cout << str << endl;
  return 0;
}
