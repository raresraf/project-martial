// Supercentral Point - Codeforces 112 A
// Clara Moraes Dantas

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  vector<pair<int, int>> points;

  for (int i = 0; i < n; i++) {
    int x, y;
    cin >> x >> y;

    points.push_back(make_pair(x, y));
  }

  int supercentral = 0;

  for (int i = 0; i < n; i++) {
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;

    for (int j = 0; j < n; j++) {
      if (j != i) {
        if (points[j].first > points[i].first &&
            points[j].second == points[i].second)
          n1++;

        else if (points[j].first < points[i].first &&
                 points[i].second == points[j].second)
          n2++;

        else if (points[i].first == points[j].first &&
                 points[j].second < points[i].second)
          n3++;

        else if (points[i].first == points[j].first &&
                 points[j].second > points[i].second)
          n4++;
      }
    }

    if (n1 > 0 && n2 > 0 && n3 > 0 && n4 > 0)
      supercentral++;
  }

  cout << supercentral << endl;

  return 0;
}
