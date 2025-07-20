#include <iostream>
using namespace std;

int main() {
  int ln, i, j;
  string str;
  cin >> str;
  ln = str.length();

  if (ln == 1) {
    cout << str;
    return 0;
  }
  for (i = 0; i < ln; i = i + 2) {
    for (j = 0; j < ln - 2; j = j + 2) {
      if (str[j] > str[j + 2]) {
        swap(str[j], str[j + 2]);
      }
    }
  }
  cout << str;

  return 0;
}