#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define mp make_pair
#define pb push_back
#define arlimit 1 << 20
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  string word;
  cin >> word;
  string hello = "hello";
  int i, j;
  for (i = 0, j = 0; i < word.size() && j < hello.size(); i++) {
    if (word[i] == hello[j])
      j++;
  }
  if (j == hello.size())
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}