#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  vector<char> v;
  for (int i = 0; i < s.size(); i++) {
    v.push_back(s[i]);
  }
  /*OR,  char s[100];
         cin >> s;

         len = strlen(s);
         vector<char>v(s,s+len);
  */
  sort(v.begin(), v.end());

  vector<char>::iterator it;
  it = unique(v.begin(), v.end());

  int curent_length = distance(v.begin(), it);

  if (curent_length % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  cout << "\n" << distance(v.begin(), it) << endl;

  for (int i = 0; i < s.size(); i++) {
    cout << v[i] << " ";
  }

  return 0;
}
