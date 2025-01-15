#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  string str;
  cin >> str;

  std::vector<int> v(26, 0);

  for (int i = 0; i < str.size(); i++) {
    v[str[i] - 'a'] = 1;
  }

  int count = 0;
  for (int i = 0; i < 26; i++) {
    if (v[i] > 0)
      count++;
  }

  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    std::cout << "IGNORE HIM!";
  return 0;
}
