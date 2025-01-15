#include <bits/stdc++.h>

using namespace std;

struct movie {
  int happy = 0;
  int satisfied = 0;
  int index = 0;
};

bool operator<(movie A, movie B) {
  if (A.happy < B.happy)
    return true;
  if (A.happy == B.happy) {
    return (A.satisfied <= B.satisfied);
  } else
    return false;
}

int main() {
  int n;
  cin >> n;
  int foo;
  vector<int> a(n + 1);
  map<int, int> lang;
  for (int i = 1; i <= n; ++i) {
    cin >> foo;
    lang[foo]++;
  }
  int m;
  cin >> m;
  vector<movie> b(m + 1);
  b[0].happy = 0;
  b[0].index = 0;
  b[0].satisfied = 0;
  for (int i = 1; i <= m; ++i) {
    cin >> foo;
    b[i].happy += lang[foo];
    b[i].index = i;
  }
  for (int i = 1; i <= m; ++i) {
    cin >> foo;
    b[i].satisfied += lang[foo];
  }

  movie best = b[1];
  for (int i = 2; i <= m; ++i) {
    if (!(b[i] < best))
      best = b[i];
  }
  cout << best.index << endl;
  return 0;
}
