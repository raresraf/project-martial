#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string input;
  cin >> input;
  input[0] = toupper(input[0]);
  cout << input << endl;
}