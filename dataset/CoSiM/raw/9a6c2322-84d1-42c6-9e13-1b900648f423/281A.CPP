#include <iostream>

using namespace std;

int main() {
  string input;
  cin >> input;
  if (input.length() > 0 && input[0] > 96) {
    input[0] = input[0] - 32;
  }
  cout << input << endl;
  return 0;
}