#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  string s;
  cin >> s;
  int count = 0;
  int one = 0, two = 0, three = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '1') {
      ++count;
      ++one;
    }
    if (s[i] == '2') {
      ++count;
      ++two;
    }
    if (s[i] == '3') {
      ++count;
      ++three;
    }
  }
  for (int i = 0; i < one; i++) {
    count -= 1;
    if (count == 0)
      cout << "1" << endl;
    else
      cout << "1+";
  }
  for (int i = 0; i < two; i++) {
    count -= 1;
    if (count == 0)
      cout << "2" << endl;
    else
      cout << "2+";
  }
  for (int i = 0; i < three; i++) {
    count -= 1;
    if (count == 0)
      cout << "3" << endl;
    else
      cout << "3+";
  }
}