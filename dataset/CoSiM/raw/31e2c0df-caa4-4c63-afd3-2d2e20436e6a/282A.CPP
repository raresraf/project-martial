#include <bits/stdc++.h>

using namespace std;
typedef std::vector<int> vi;
typedef long long ll;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int x = 0;
  for (int i = 0; i < n; i++) {
    string str;
    cin >> str;
    if (str[0] == '+')
      x++;
    else if (str[0] == '-')
      x--;
    else if (str[1] == '-')
      x--;
    else
      x++;
  }
  cout << x;
  return 0;
}
