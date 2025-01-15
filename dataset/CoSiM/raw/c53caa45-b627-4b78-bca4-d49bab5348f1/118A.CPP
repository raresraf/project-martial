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
  string vowels = "aeiouyAEIOUY";
  string word;
  cin >> word;
  vector<char> myvec;
  for (int i = 0; i < word.size(); i++) {
    bool found = false;
    for (int j = 0; j < vowels.size(); j++) {
      if (word[i] == vowels[j])
        found = true;
    }
    if (!found) {
      myvec.pb('.');
      myvec.pb(tolower(word[i]));
    }
  }
  for (int i = 0; i < myvec.size(); i++)
    cout << myvec[i];
}