#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string input;
  cin >> input;
  auto it = input.begin();
  (*it) = toupper(*it);

  cout << input;

  return 0;
}
