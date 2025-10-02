#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ceres/ceres.h>
#include <fstream>

struct CircleCenter {
    int frame_number;
    double x;
    double y;
    double timestamp;
};

// 检测蓝色圆形的函数
bool detectBlueCircle(const cv::Mat& frame, cv::Point2d& center) {
    // 转换到HSV颜色空间，更好地检测蓝色
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // 定义蓝色的HSV范围
    cv::Scalar lower_blue(10, 50, 50);   // 蓝色下限
    cv::Scalar upper_blue(150, 255, 255);  // 蓝色上限
    
    // 创建蓝色掩码
    cv::Mat blue_mask;
    cv::inRange(hsv, lower_blue, upper_blue, blue_mask);
    
    // 形态学操作去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE, kernel);
    
    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blue_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return false;
    }
    
    // 找到最大的轮廓（假设蓝色圆形是最大的蓝色物体）
    double max_area = 0;
    int max_index = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_index = i;
        }
    }
    
    
    // 使用最小外接圆来获取圆心和半径
    cv::Point2f circle_center;
    float radius;
    cv::minEnclosingCircle(contours[max_index], circle_center, radius);
    
    center.x = circle_center.x;
    center.y = 720-circle_center.y;
    
    return true;
}

// 存储坐标数据的结构体
struct PositionData {
    double t;  // 时间 (s)
    double x;  // x坐标 (px)
    double y;  // y坐标 (px)
};

// 运动模型结构体，用于Ceres优化
struct ProjectileMotionCost {
    ProjectileMotionCost(double t, double x, double y, double x0, double y0)
        : t(t), x(x), y(y), x0(x0), y0(y0) {}
    
    template<typename T>
    bool operator()(const T* const parameters, T* residual) const {
        // 参数: [vx0, vy0, g, k]
        const T& vx0 = parameters[0];
        const T& vy0 = parameters[1];
        const T& g = parameters[2];
        const T& k = parameters[3];
        
        const T delta_t = T(t);
        
        // x方向运动方程: x(t) = x0 + (vx0/k) * (1 - exp(-k*Δt))
        T x_pred = T(x0) + (vx0 / k) * (T(1.0) - ceres::exp(-k * delta_t));
        
        // y方向运动方程: y(t) = y0 + ((vy0 + g/k)/k) * (1 - exp(-k*Δt)) - (g/k)*Δt
        T y_pred = T(y0) + ((vy0 + g / k) / k) * (T(1.0) - ceres::exp(-k * delta_t)) - (g / k) * delta_t;
        
        // 计算残差
        residual[0] = x_pred - T(x);
        residual[1] = y_pred - T(y);
        
        return true;
    }
    
private:
    const double t, x, y, x0, y0;
};

// 读取坐标数据
std::vector<PositionData> readPositionData(const std::string& filename) {
    std::vector<PositionData> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return data;
    }
    
    std::string line;
    // 跳过标题行
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        PositionData pos;
        
        // 格式: Frame,Time(s),X,Y
        std::getline(ss, token, ',');  // 跳过帧号
        std::getline(ss, token, ','); pos.t = std::stod(token);
        std::getline(ss, token, ','); pos.x = std::stod(token);
        std::getline(ss, token, ','); pos.y = std::stod(token);
        
        data.push_back(pos);
    }
    
    file.close();
    return data;
}

int main() {

    // 获取视频文件路径
    std::string videoPath = "../videos/video.mp4";

    // 创建VideoCapture对象
    cv::VideoCapture cap(videoPath);

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cout << "错误：无法打开视频文件: " << videoPath << std::endl;
        return -1;
    }

    // 获取视频信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // std::cout << "视频信息:" << std::endl;
    // std::cout << " - 分辨率: " << width << " x " << height << std::endl;
    // std::cout << " - 帧率: " << fps << " FPS" << std::endl;
    // std::cout << " - 总帧数: " << totalFrames << std::endl;
    // std::cout << std::endl;

    // 存储圆心坐标的数组
    std::vector<CircleCenter> circle_positions;

    cv::Mat frame;
    int frameCount = 0;

    // 创建显示窗口
    // cv::namedWindow("Video Frame", cv::WINDOW_AUTOSIZE);

    // std::cout << "开始读取视频帧... (按ESC键退出)" << std::endl;

    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            break;
        }

        frameCount++;
        double timestamp = frameCount / fps;  // 计算时间戳（秒）

        cv::Point2d center;
        bool circle_detected = detectBlueCircle(frame, center);

        if (circle_detected) {
            // 保存圆心坐标
            CircleCenter pos;
            pos.frame_number = frameCount;
            pos.x = center.x;
            pos.y = center.y;
            pos.timestamp = timestamp;
            circle_positions.push_back(pos);

            // 在图像上绘制圆心和轨迹
            cv::circle(frame, center, 5, cv::Scalar(0, 255, 0), -1);  // 绿色圆心点
            cv::circle(frame, center, 20, cv::Scalar(0, 255, 0), 2);  // 绿色圆形轮廓
            
            // 绘制坐标文本
            std::string coord_text = "(" + std::to_string((int)center.x) + ", " + 
                                    std::to_string((int)center.y) + ")";
            cv::putText(frame, coord_text, 
                       cv::Point(center.x + 10, center.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            // 绘制帧信息
            std::string frame_info = "Frame: " + std::to_string(frameCount) + 
                                   " Time: " + std::to_string(timestamp).substr(0, 5) + "s";
            cv::putText(frame, frame_info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

            // std::cout << "帧 " << frameCount << " - 圆心坐标: (" 
            //           << center.x << ", " << center.y << ")" << std::endl;
        } else {
            // std::cout << "帧 " << frameCount << " - 未检测到蓝色圆形" << std::endl;
        }

        // 显示处理后的帧
        // cv::imshow("Blue Circle Tracking", frame);

        // 按ESC退出
        // char key = cv::waitKey(1000 / 60);  // 大约60fps的显示速度
        // if (key == 27) {
        //     std::cout << "用户中断处理!" << std::endl;
        //     break;
        // }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();

    // 输出统计信息
    // std::cout << "\n=== 处理完成 ===" << std::endl;
    // std::cout << "总帧数: " << frameCount << std::endl;
    // std::cout << "检测到圆形的帧数: " << circle_positions.size() << std::endl;
    // std::cout << "检测成功率: " << 
    //     (static_cast<double>(circle_positions.size()) / frameCount * 100) << "%" << std::endl;

    // std::cout << "\n将圆心坐标保存到文件: circle_positions.txt" << std::endl;
    std::ofstream outfile("circle_positions.txt");
    outfile << "Frame,Time(s),X,Y\n";
    for (const auto& pos : circle_positions) {
        outfile << pos.frame_number << "," << pos.timestamp << "," 
                << pos.x << "," << pos.y << "\n";
    }
    outfile.close();




    std::string data_file = "circle_positions.txt";
    std::vector<PositionData> positions = readPositionData(data_file);
    
    if (positions.empty()) {
        std::cerr << "没有读取到数据或文件为空" << std::endl;
        return -1;
    }
    
    // std::cout << "成功读取 " << positions.size() << " 个数据点" << std::endl;
    
    // 设置初始位置 (使用第一帧数据)
    double x0 = positions[0].x;
    double y0 = positions[0].y;
    double t0 = positions[0].t;
    
    // 调整时间，使第一帧时间为0
    for (auto& pos : positions) {
        pos.t -= t0;
    }
    
    // std::cout << "初始位置: (" << x0 << ", " << y0 << ")" << std::endl;
    
    // 初始参数估计
    double initial_parameters[4] = {0.0, 0.0, 500.0, 0.1};  // [vx0, vy0, g, k]
    
    // 设置参数边界
    ceres::Problem problem;
    
    // 参数边界约束
    problem.AddParameterBlock(initial_parameters, 4,nullptr);
    
    // // 设置参数边界: g ∈ [100, 1000], k ∈ [0.01, 1]
    // ceres::SubsetParameterization* parameterization = 
    //     new ceres::SubsetParameterization(4, {2, 3});  // 只对g和k设置边界
    
    // problem.SetParameterization(initial_parameters, parameterization);
    
    // 添加边界约束
    problem.SetParameterLowerBound(initial_parameters, 2, 100.0);   // g下限
    problem.SetParameterUpperBound(initial_parameters, 2, 1000.0);  // g上限
    problem.SetParameterLowerBound(initial_parameters, 3, 0.01);    // k下限
    problem.SetParameterUpperBound(initial_parameters, 3, 1.0);     // k上限
    
    // 为每个数据点添加残差块
    for (const auto& pos : positions) {
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<ProjectileMotionCost, 2, 4>(
                new ProjectileMotionCost(pos.t, pos.x, pos.y, x0, y0));
        problem.AddResidualBlock(cost_function, nullptr, initial_parameters);
    }
    
    // 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    
    // 求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.FullReport() << std::endl;
    
    // 输出拟合结果
    double vx0 = initial_parameters[0];
    double vy0 = initial_parameters[1];
    double g = initial_parameters[2];
    double k = initial_parameters[3];
    
    std::cout << "\n=== 拟合结果 ===" << std::endl;
    std::cout << "初始速度 vx0: " << vx0 << " px/s" << std::endl;
    std::cout << "初始速度 vy0: " << vy0 << " px/s" << std::endl;
    std::cout << "合速度: " << std::sqrt(vx0*vx0 + vy0*vy0) << " px/s" << std::endl;
    std::cout << "发射角度: " << std::atan2(vy0, vx0) * 180.0 / M_PI << " 度" << std::endl;
    std::cout << "重力加速度 g: " << g << " px/s²" << std::endl;
    std::cout << "阻尼系数 k: " << k << " 1/s" << std::endl;
    
    // 计算拟合误差
    double total_error_x = 0.0;
    double total_error_y = 0.0;
    double max_error = 0.0;
    
    // std::cout << "\n=== 误差分析 ===" << std::endl;
    for (const auto& pos : positions) {
        double delta_t = pos.t;
        
        // 计算预测值
        double x_pred = x0 + (vx0 / k) * (1 - std::exp(-k * delta_t));
        double y_pred = y0 + ((vy0 + g / k) / k) * (1 - std::exp(-k * delta_t)) - (g / k) * delta_t;
        
        double error_x = std::abs(x_pred - pos.x);
        double error_y = std::abs(y_pred - pos.y);
        double distance_error = std::sqrt(error_x * error_x + error_y * error_y);
        
        total_error_x += error_x;
        total_error_y += error_y;
        max_error = std::max(max_error, distance_error);
        
        // std::cout << "t=" << delta_t << "s: 实际(" << pos.x << ", " << pos.y 
        //           << ") 预测(" << x_pred << ", " << y_pred 
        //           << ") 误差=" << distance_error << " px" << std::endl;
    }
    
    double avg_error_x = total_error_x / positions.size();
    double avg_error_y = total_error_y / positions.size();
    double avg_distance_error = std::sqrt(avg_error_x * avg_error_x + avg_error_y * avg_error_y);
    
    // std::cout << "\n平均误差 - X: " << avg_error_x << " px, Y: " << avg_error_y << " px" << std::endl;
    // std::cout << "平均距离误差: " << avg_distance_error << " px" << std::endl;
    // std::cout << "最大距离误差: " << max_error << " px" << std::endl;
    
    // 计算相对误差百分比
    double x_range = 0.0, y_range = 0.0;
    for (const auto& pos : positions) {
        x_range = std::max(x_range, std::abs(pos.x - x0));
        y_range = std::max(y_range, std::abs(pos.y - y0));
    }
    
    double relative_error_x = (avg_error_x / x_range) * 100;
    double relative_error_y = (avg_error_y / y_range) * 100;
    double overall_relative_error = (avg_distance_error / std::sqrt(x_range*x_range + y_range*y_range)) * 100;
    
    std::cout << "\n=== 相对误差 ===" << std::endl;
    std::cout << "X方向相对误差: " << relative_error_x << "%" << std::endl;
    std::cout << "Y方向相对误差: " << relative_error_y << "%" << std::endl;
    std::cout << "总体相对误差: " << overall_relative_error << "%" << std::endl;
    
    // if (overall_relative_error <= 3.0) {
    //     std::cout << "\n✅ 拟合成功！误差在3%以内" << std::endl;
    // } else {
    //     std::cout << "\n❌ 拟合误差超过3%，可能需要调整初始参数或检查数据" << std::endl;
    // }
    
    return 0;
}