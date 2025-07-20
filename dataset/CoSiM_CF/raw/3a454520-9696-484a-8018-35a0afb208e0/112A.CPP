#include <iostream>

using namespace std;

int main() {
  string s1 = "";
  string s2 = "";
  int result = 0;
  cin >> s1 >> s2;
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
  }

  if (s1 == s2) {
    result = 0;
  } else if (s1 > s2) {
    result = 1;
  } else if (s1 < s2) {
    result = -1;
  }

  cout << result << endl;

  return 0;
}
