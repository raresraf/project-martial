#include <bits/stdc++.h>
using namespace std;
#define f first
#define mp make_pair
#define pb push_back
#define arlimit 1000000
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  string name;
  cin >> name;
  int ary[26] = {0};
  for (int i = 0; i < name.size(); i++) {
    int to_int = name[i] - 97;
    ary[to_int] = 1;
  }
  int sum = 0;
  for (int i = 0; i < 26; i++)
    sum += ary[i];
  if (sum % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}