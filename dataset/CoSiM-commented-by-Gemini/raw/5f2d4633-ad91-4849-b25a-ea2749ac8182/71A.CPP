#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;

  while (n--) {
    string input;
    cin >> input;
    int length = input.length();

    if (length > 10)
      cout << input[0] << length - 2 << input[length - 1] << endl;
    else
      cout << input << endl;
  }

  return 0;
}