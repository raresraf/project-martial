#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int suma = 0;
  string s;

  for (int i = 1; i <= n; i++) {
    cin >> s;

    if (s == "++X") {
      suma++;
    }
    if (s == "X++") {
      suma++;
    }
    if (s == "--X") {
      suma--;
    }
    if (s == "X--") {
      suma--;
    }
  }

  cout << suma;

  return 0;
}
