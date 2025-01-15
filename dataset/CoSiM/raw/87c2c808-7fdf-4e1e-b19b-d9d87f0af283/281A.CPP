#include <cctype>
#include <iostream>
using namespace std;

int main() {
  char input[1000];
  cin >> input;

  input[0] = toupper(input[0]);

  cout << input << endl;

  return 0;
}