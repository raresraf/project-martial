#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <set>
#include <vector>

std::vector<int> vec;
std::vector<int> vec2;

int main() {
    int sum = 0;
    for (int i = 1; true; ++i) {
        sum += i;
        if (sum > 2000) {
            break;
        }
        vec.push_back(sum);
        vec2.push_back(i);
    }

    // std::sort(vec.begin(), vec.end());
    // for (int i = 0; i < (int)vec.size(); ++i) {
    //     printf("[%d] %d %d\n", i, vec2[i], vec[i]);
    // }
    // printf("vec.size() = %d\n", (int)vec.size());

    int n;
    scanf("%d", &n);
    int m = -1;
    for (int i = 1; i < (int)vec.size(); ++i) {
        if (n < vec[i]) {
            m = i - 1;
            break;
        }
    }

    n -= vec[m];
    while (n > 0) {
        for (int i = m; i >= 0 && n > 0; --i) {
            vec2[i] += 1;
            n -= 1;
        }
    }

    printf("%d\n", m + 1);
    for (int i = 0; i <= m; ++i) {
        printf("%d ", vec2[i]);
    }
    printf("\n");

    return 0;
}
