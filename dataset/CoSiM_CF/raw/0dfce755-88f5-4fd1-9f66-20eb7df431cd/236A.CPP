#include <cstring>
#include <iostream>

using namespace std;

int main() {
  string x;

  cin >> x;
  int count;

  count = x.length();
  for (int i = 0; i <= x.length(); i++) {
    for (int j = i + 1; j <= x.length(); j++) {
      if (x[i] == x[j]) {
        count--;
        break;
      }
    }
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
