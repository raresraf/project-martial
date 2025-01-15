#include <iostream>
#include <stdlib.h>
#include <string.h>

using namespace std;

int main() {

  int n = 0;
  int x = 0;
  string s = "";

  cin >> n;
  cin.ignore();

  for (int i = 0; i < n; ++i) {
    getline(cin, s);
    if (s.length() == 3) {
      if (s == "++X" || s == "X++") {
        x += 1;
      } else if (s == "--X" || s == "X--") {
        x -= 1;
      }
    } else {
      exit(0);
    }
  }

  cout << endl << x;

  return 0;
}
