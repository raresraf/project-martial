#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<string> vs;
typedef pair<int, int> pi;
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;

#define sz(v) ((int)((v).size()))
#define all(v) ((v).begin(), (v).end())
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
#define nl << "\n"

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int X = 0, N;
  string op;
  cin >> N;
  for (int i = 0; i < N; i++) {
    cin >> op;
    if (op == "X++")
      X++;
    else if (op == "X--")
      X--;
    else if (op == "++X")
      ++X;
    else if (op == "--X")
      --X;
  }
  cout << X << endl;
}
