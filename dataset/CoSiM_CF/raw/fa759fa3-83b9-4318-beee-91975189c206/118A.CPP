#include <bits/stdc++.h>

using namespace std;
typedef std::vector<int> vi;
typedef long long ll;

int main() {
  string str;
  cin >> str;

  for (int i = 0; i < str.size(); i++) {
    str[i] = tolower(str[i]);
    char ch = '.';
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y') {
      str.erase(str.begin() + i);
      i--;
    } else {
      str.insert(str.begin() + i, 1, ch);
      i++;
    }
  }

  cout << str;

  return 0;
}
