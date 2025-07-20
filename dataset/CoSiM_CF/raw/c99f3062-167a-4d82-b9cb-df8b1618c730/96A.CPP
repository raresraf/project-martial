#include <bits/stdc++.h>

using namespace std;
typedef std::vector<int> vi;
typedef long long ll;

int main(int argc, char const *argv[]) {
  string str;
  cin >> str;

  char ch = str[0];
  int flag = 0;
  int count = 0;
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == ch)
      count++;
    else {
      count = 1;
      ch = str[i];
    }
    if (count == 7) {
      flag = 1;
      break;
    }
  }

  if (flag == 1)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
