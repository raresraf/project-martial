#include <bits/stdc++.h>
#include <string>

using namespace std;

typedef long long ll;
typedef vector<ll> vi;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  string str;
  vi A(n);
  for (int i = 0; i < n; i++) {
    scanf("%lld", &A[i]);
  }
  while (n--) {
    cin >> str;
    if (str.size() < 11)
      cout << str << endl;
    else {
      string str1;
      str1 += str[0];
      str1 = str1 + to_string(str.size() - 2);
      str1 += str[str.size() - 1];
      cout << str1 << endl;
    }
  }
  return 0;
}
