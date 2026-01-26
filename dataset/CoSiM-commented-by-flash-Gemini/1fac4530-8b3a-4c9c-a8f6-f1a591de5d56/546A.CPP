#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

int N, K;
pair<int, int> chapters[100010];

int main() {
  cin >> N;
  for (int i = 0; i < N; i++)
    cin >> chapters[i].first >> chapters[i].second;
  cin >> K;
  int nryet = N;
  for (int i = 0; i < N; i++) {
    if (K > chapters[i].second) {
      nryet--;
    }
  }
  cout << nryet << endl;
  return 0;
}