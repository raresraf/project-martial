#include <iostream>
#include <vector>
using namespace std;
typedef long long int ll;

vector<int> seq;
vector<bool> pos;
int N;

ll calc() {
  int j;
  ll ret = 0;
  int mx;
  for (int i = 0; i < N; i++) {
    j = i + 1;
    mx = seq[i];
    while (j < N && pos[j] == pos[i]) {
      mx = max(mx, seq[j]);
      j++;
    }
    ret += mx;
    i = j - 1;
  }
  return ret;
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  int T;
  cin >> T;
  while (T--) {
    cin >> N;
    seq.resize(N);
    pos.resize(N);
    for (int n = 0; n < N; n++) {
      cin >> seq[n];
      pos[n] = (seq[n] > 0);
    }
    cout << calc() << "\n";
  }
  return 0;
}
