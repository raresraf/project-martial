#include <bits/stdc++.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main() {
  vector<int> v; // Using this vector,the score of the participants are stored.
  int c = 0, n, t, k;
  scanf("%d", &n);
  cin >> k;
  for (int i = 0; i < n; i++) {
    cin >> t;
    v.push_back(t);
  }
  int k1 = v[k];
  if (k1 <= 0) { // According to teh condition,the score should not be<=0
    std::cout << "0" << std::endl;
    exit(0);
  }
  sort(v.begin(), v.end()); // if it isn't ,sort the array in ascending order

  auto i =
      lower_bound(v.begin(), v.end(), k1); // lower bound means all values >=k
  cout << v.end() - i; // prins the no. of elements >=k,since v.end() points to
                       // the end of the vector
}

// Problem:https://codeforces.com/problemset/problem/158/A
// I didn't get all the test cases,feel free to modify
