#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main(void) {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  int n;
  cin >> n;
  while (n--) {
    string word;
    cin >> word;
    if (word.size() > 10)
      cout << word[0] << word.size() - 2 << word[word.size() - 1] << endl;
    else
      cout << word << endl;
  }
}