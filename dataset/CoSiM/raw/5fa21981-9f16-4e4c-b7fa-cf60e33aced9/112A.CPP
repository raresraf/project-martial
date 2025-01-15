#include <iostream>
#include <string>

using namespace std;

int main() {
  string str1, str2;
  cin >> str1;
  cin >> str2;

  for (int i = 0; i < str1.length(); i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);
  }

  int c1 = 0;
  int c2 = 0;

  for (int i = 0; i < str1.length(); i++) {
    c1 += int(str1[i]) - 96;
    c2 += int(str2[i]) - 96;

    cout << c1 << endl;
    cout << c2 << endl;
  }

  if (c1 > c2) {
    cout << 1;
  } else if (c2 > c1) {
    cout << -1;
  } else if (c1 == c2) {
    cout << 0;
  }

  return 0;
}
