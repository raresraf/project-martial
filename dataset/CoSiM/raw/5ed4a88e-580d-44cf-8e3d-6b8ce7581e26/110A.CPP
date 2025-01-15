#include <iostream>
#include <string>
using namespace std;

int main() {
  string kata;
  int total = 0;
  bool dis = true, a = false, b = false;
  cin >> kata;

  for (int i = 0; i < kata.length(); i++) {
    if (kata[i] == '4' || kata[i] == '7') {
      total++;
      if (total > 7) {
        break;
      }
    }
  }

  if (total == 4 || total == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
