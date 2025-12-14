#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, count1 = 1, count2 = 0;
  cin >> n;
  vector<string> s;
  string word;

  for (int i = 0; i < n; i++) {
    cin >> word;
    s.push_back(word);
    if (s[i].size() > 10) {
      cout << s[i][0] << s[i].size() - 2 << s[i][s[i].size() - 1] << endl;
      // for(int j = 0; j < s[i].size(); j++){}
    } else
      cout << s[i] << endl;
  }

  return 0;
}
