#include <algorithm>
#include <iostream>

using namespace std;
int main() {

  string s;
  int a;
  cin >> s;

  sort(s.begin(), s.end());
  int count = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != s[i + 1]) {
      count++;
    }
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
