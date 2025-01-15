#include <iostream>
#include <string>

using namespace std;

int main() {
  string input;
  getline(cin, input);

  int a[input.length()];

  for (int i = 0; i < input.length(); i++) {
    a[i] = int(input.at(i)) - 48;
  }

  for (int i = 0; (i + 6) < input.length(); i++) {
    if (a[i] == a[i + 1] && a[i] == a[i + 2] && a[i] == a[i + 3] &&
        a[i] == a[i + 4] && a[i] == a[i + 5] && a[i] == a[i + 6]) {
      cout << "YES";
      return 0;
    }
  }

  cout << "NO";
  return 0;
}
