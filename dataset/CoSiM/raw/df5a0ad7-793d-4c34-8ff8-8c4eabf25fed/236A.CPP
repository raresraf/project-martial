// Codeforces
// 236A - Boy or girl

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  string s;
  cin >> s;
  vector<char> chars;

  for (unsigned int i = 0; i < s.size(); i++) {
    if (count(chars.begin(), chars.end(), s[i]) == 0)
      chars.push_back(s[i]);
  }

  if (chars.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";

  return 0;
}
