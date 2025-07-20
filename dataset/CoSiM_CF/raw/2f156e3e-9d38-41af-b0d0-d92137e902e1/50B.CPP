#include <algorithm>
#include <bitset>
#include <climits>
#include <complex>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <math.h>
#include <memory.h>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <string>
#include <vector>

#define F first
#define S second
#define endl "\n"
#define mp make_pair
#define pb push_back
#define MOD 1000000007
#define pi 3.141592653589793
#define y1 zjdfshnvoavaofobiopj
using namespace std;

long long cnt, found[301];
string st;

int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  cin >> st;
  for (int i = 0; i < st.size(); i++)
    ++found[st[i]];
  for (int i = 0; i < 300; i++)
    cnt += found[i] * (found[i] - 1);
  cout << cnt + st.size() << endl;

  return 0;
}