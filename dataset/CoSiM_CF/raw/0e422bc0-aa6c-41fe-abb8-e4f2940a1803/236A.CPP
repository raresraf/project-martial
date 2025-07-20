#include <iostream>
#include <vector>

using namespace std;

int main() {
  string s;
  vector<int> a(26, 0);
  cin >> s;
  for (int i = 0; i < s.size(); ++i)
    a[s[i] - 'a'] = 1;

  int count = 0;

  for (int i = 0; i < 26; ++i)
    count += a[i];

  if (count % 2)
    cout << "IGNORE HIM!\n";
  else
    cout << "CHAT WITH HER!\n";
}