#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <deque>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <search.h>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

char letters[27] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                    's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
char reversed_letters[27] = {'z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r',
                             'q', 'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i',
                             'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'};
char vowels[13] = {'A', 'O', 'E', 'Y', 'U', 'I', 'a', 'o', 'y', 'e', 'u', 'i'};
int find_max_vector(vector<int> arr) {
  int ans = 0;
  for (int i = 0; i < arr.size(); i++) {
    if (arr[i] > ans) {
      ans = arr[i];
    }
  }
  return ans;
}

int linear_count(int n, vector<int> arr) {
  int ans = 0;
  for (int i = 0; i < arr.size(); i++) {
    if (arr[i] == n) {
      ans++;
    }
  }
  return ans;
}

int find_min_vector(vector<int> arr) {
  bool move = true;
  int ans = 0;
  for (int i = 0; i < arr.size(); i++) {
    if (move) {
      ans = arr[i];
    } else if (arr[i] < ans) {
      ans = arr[i];
    }
  }
  return ans;
}

bool is_palindrome(string s) {
  string copys = s;
  reverse(copys.begin(), copys.end());
  return copys == s;
}

bool exist(char arr[13], char s) {
  for (int i = 0; i < 12; i++) {
    if (arr[i] == s) {
      return true;
    }
  }
  return false;
}

bool linear_search_boolean_string(string str, char s) {
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == s) {
      return true;
    }
  }
  return false;
}

int sum_vector(vector<int> arr) {
  int ans = 0;
  for (int i = 0; i < arr.size(); i++) {
    ans += arr[i];
  }
  return ans;
}

string lower(string str) {
  for (int i = 0; i < str.size(); i++) {
    str[i] = tolower(str[i]);
  }
  return str;
}

string upper(string str) {
  for (int i = 0; i < str.size(); i++) {
    str[i] = toupper(str[i]);
  }
  return str;
}

int main() {
  pair<int, int> p = {0, 0};
  int a = 0, b = 0, c = 0, d, n, k, test, ans = 0, pointer = 0;
  int kk[101];
  vector<int> arr, cs;
  map<int, int> frq;
  frq['R'] = 0;
  frq['L'] = 0;
  unordered_set<char> sett;
  set<int> ss;
  bool move = true;
  string str, s1 = "", s2;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> k;
    arr.push_back(k);
  }
  if (arr.size() == 1) {
    ans = 1;
  } else {
    sort(arr.begin(), arr.end());
    while (a <= sum_vector(arr)) {
      a += arr[arr.size() - 1];
      arr.erase(arr.end() - 1);
      ans++;
    }
  }
  cout << ans << endl;
  return 0;
}
