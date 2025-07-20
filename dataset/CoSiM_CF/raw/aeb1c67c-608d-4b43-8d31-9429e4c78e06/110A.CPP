#include <iostream>

using namespace std;

int main() {
  string n = "";
  string res = "";
  int counter = 0;
  cin >> n;
  for (int i = 0; i < n.length(); i++) {
    if (n[i] == '4' || n[i] == '7') {
      counter++;
    }
  }
  if (counter == 0) {
    res = "NO";
  }

  while (counter != 0) {
    if (counter % 10 != 4 && counter % 10 != 7) {
      res = "NO";
      break;
    } else {
      res = "YES";
    }
    counter /= 10;
  }

  cout << res << endl;

  return 0;
}
