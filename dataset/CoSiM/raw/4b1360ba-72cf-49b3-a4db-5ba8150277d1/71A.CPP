#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  string input;
  cin >> n;
  while (n-- > 0) {
    cin >> input;

    if (input.length() > 10)
      cout << input.front() << input.length() - 2 << input.back() << endl;
    else
      cout << input << endl;
  }
  return 0;
}