#include <iostream>
using namespace std;
int main() {

  string n;
  int lDigit = 0;
  cin >> n;

  for (int i = 0; i < n.length(); i++) {
    if (n[i] == '4' || n[i] == '7') {
      lDigit++;
    }
  }

  if (lDigit == 7 || lDigit == 4) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
